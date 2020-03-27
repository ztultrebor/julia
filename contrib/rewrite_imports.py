import pefile, shutil, os, struct
from optparse import OptionParser

## Theory of operation:
#
# We need to rewrite a DLL import within a PE file with a (typically much longer) string.
# The strings for a DLL import are referred to by RVA (relative virtual address), and as
# such are trivially redirected.  `.idata` (the import data section) structures contain
# these RVA values and point to NULL-separated strings embedded at the end of the `.idata`
# section.  Overwriting these strings is technically feasible, but we run into issues
# when the length of strings overflows the `.idata` section size; if we must increase it,
# we may need to re-arrange the virtual address spaces the sections are loaded into the
# process space at, and that may require re-linking the executable, as it is possible to
# have cross-section RVA references.  So instead, we turn a liability into an asset, and
# use cross-section RVAs in the `Name` field of the import descriptor in `.idata`.  We
# allocate a new section (potentially expanding the header block if we don't have enough
# space for our new section header), and dump all our new strings into the new section.
# Note that there is often data at the end of a PE file that is not in any section, so
# we must move that forward in order to fit our new section in.  Then, we calculate the
# appropriate RVAs to point to these new strings from the import descriptor structures.

## Caveats:
# 
# This works just fine to load images on Windows, however some tools such as `objdump`
# take a moral exception to cross-section references.  They silently fail when printing
# out PE import descriptor structures that we have modified, thinking that they have
# read in junk/corrupted PE images.

parser = OptionParser(usage="usage: %prog [--rewrite FROM:TO]... files")
parser.add_option("--rewrite", dest="rewrites", metavar="FROM:TO",
                  action="append", help="Rewrite an import library FROM to TO")
parser.add_option("-v", "--verbose", dest="verbose", action="store_true")

options, args = parser.parse_args()
if len(args) < 1:
    parser.error("Must provide PE files to modify!")

# Extract options
verbose = options.verbose

# Extract rewrites
rewrites = [r.split(":") for r in options.rewrites]
# Drop no-op rewrites, then quit gracefully if we have nothing to do
rewrites = [r for r in rewrites if r[0] != r[1]]
if not rewrites:
    if verbose:
        print("no substantive relocations requested, exiting cleanly")
    sys.exit(0)

def rewrite_pe_file(fname):
    pe = pefile.PE(fname)

    # Let's leave modifying a `.rewrote` section for the future.
    if [s for s in pe.sections if s.Name == b'.rewrote']:
        parser.error("Cannot (yet) run this tool on the same executable twice!")
    print(f"Rewriting {fname}")

    # Find last section on disk, to figure out where to place our new section in the file
    last_disk_sec = max(pe.sections, key=lambda s: s.PointerToRawData + s.SizeOfRawData)

    # Find last section in memory, to figure out where to place our new section in memory
    last_vm_sec = max(pe.sections, key=lambda s:s.VirtualAddress + s.Misc_VirtualSize)

    # Generate a new section with a header and the string representations of our names:
    sizeof_section_header = 40
    new_strings = [new.encode() for (old, new) in rewrites]
    new_string_table = b'\x00\x00'.join(new_strings) + b'\x00\x00'

    new_virtual_address = pe.adjust_SectionAlignment(
        last_vm_sec.VirtualAddress + last_vm_sec.Misc_VirtualSize + pe.OPTIONAL_HEADER.SectionAlignment - 1,
        pe.OPTIONAL_HEADER.SectionAlignment,
        pe.OPTIONAL_HEADER.FileAlignment,
    )
    new_disk_address = pe.adjust_FileAlignment(
        last_disk_sec.PointerToRawData + last_disk_sec.SizeOfRawData + pe.OPTIONAL_HEADER.FileAlignment - 1,
        pe.OPTIONAL_HEADER.FileAlignment,
    )
    new_disk_size = pe.adjust_FileAlignment(
        len(new_string_table) + pe.OPTIONAL_HEADER.FileAlignment - 1,
        pe.OPTIONAL_HEADER.FileAlignment,
    )
    jheader = struct.pack(
        '8sIIIIIIHHI',
        b'.rewrote',
        # Create a section of this size in memory, at this address
        len(new_string_table),
        new_virtual_address,
        # Store it in a file chunk of this size, at this offset within the file
        new_disk_size,
        new_disk_address,
        0,
        0,
        0,
        0,
        0xC0000040, # INITIALIZED_DATA | MEM_READ | MEM_WRITE,
    )

    # Also calculate the RVA offsets for each string we just added in:
    new_string_lens = [len(n) + 2 for n in new_strings]
    rva_idxs = [sum(new_string_lens[:idx]) for idx in range(len(new_strings))]
    rva_offsets = {new_strings[idx]:(new_virtual_address + rva_idxs[idx]) for idx in range(len(new_strings))}

    # Do we need to expand the headers to make way for our new section header?
    section_table_start = pe.DOS_HEADER.e_lfanew + pe.FILE_HEADER.sizeof() + pe.NT_HEADERS.sizeof() + pe.FILE_HEADER.SizeOfOptionalHeader
    section_table_end = pe.DOS_HEADER.e_lfanew + pe.FILE_HEADER.sizeof() + pe.NT_HEADERS.sizeof() + pe.FILE_HEADER.SizeOfOptionalHeader + pe.FILE_HEADER.NumberOfSections * sizeof_section_header
    old_sizeof_headers = pe.OPTIONAL_HEADER.SizeOfHeaders
    expand_headers_amnt = 0
    while pe.OPTIONAL_HEADER.SizeOfHeaders - section_table_end <= sizeof_section_header:
        expand_headers_amnt += pe.OPTIONAL_HEADER.FileAlignment
    if expand_headers_amnt > 0:
        print(f"Must expand headers block by 0x{expand_headers_amnt:X} to fit new section headers")

    # Everything after this point in the file is out-of-image data, we'll need to move it.
    extra_data_start = last_disk_sec.PointerToRawData + last_disk_sec.SizeOfRawData

    # Increment number of sections and overall image size
    pe.FILE_HEADER.NumberOfSections += 1
    pe.OPTIONAL_HEADER.SizeOfImage += pe.OPTIONAL_HEADER.SectionAlignment
    pe.OPTIONAL_HEADER.SizeOfInitializedData += new_disk_size

    # check to see if certain pointers in the header need to be updated:
    pe.FILE_HEADER.PointerToSymbolTable += expand_headers_amnt
    if pe.FILE_HEADER.PointerToSymbolTable >= last_disk_sec.PointerToRawData + last_disk_sec.SizeOfRawData:
        # If the symbol table was in an overlay, push it forward:
        pe.FILE_HEADER.PointerToSymbolTable += new_disk_size

    # Generate RVAs that point to our new section for these import names
    import_descs = [s for s in pe.__structures__ if 'IMAGE_IMPORT_DESCRIPTOR' in s.name]
    for d in import_descs:
        curr_name = pe.get_string_at_rva(d.Name)
        for (old, new) in rewrites:
            if curr_name == old.encode():
                # Calculate RVA from this section to our new section:
                if verbose:
                    print(f" - {old} -> {new} via RVA change 0x{d.Name:X} -> 0x{rva_offsets[new.encode()]:X}")
                d.Name = rva_offsets[new.encode()]
                

    # Save to a `.new` file
    new_fname = fname[:-4] + "-new.exe"
    pe.write(new_fname)
    pe.close()

    # Jam our new jdata section into the `.new` file, moving all extra data out of the way:
    with open(new_fname, "r+b") as f:
        # First, if we need to expand the headers, do so
        if expand_headers_amnt > 0:
            f.seek(old_sizeof_headers)
            rest_of_file = f.read()
            f.write(bytearray(expand_headers_amnt))
            f.write(rest_of_file)
        
        # Next, write out our beautiful new header
        f.seek(section_table_end)
        f.write(jheader)

        # First, read the extra data in
        f.seek(extra_data_start + expand_headers_amnt)
        extra_data = f.read()
        
        # Zoom back and insert new string table, then write the extra data out again
        f.seek(new_disk_address + expand_headers_amnt)
        f.write(new_string_table)
        f.write(bytearray(new_disk_size - len(new_string_table)))
        f.write(extra_data)

    os.remove(fname)
    shutil.move(new_fname, fname)

for fname in args:
    rewrite_pe_file(fname)