target_read_memory (bfd_vma memaddr,

                    bfd_byte *myaddr,

                    int length,

                    struct disassemble_info *info)

{

    int i;

    for(i = 0; i < length; i++) {

        myaddr[i] = ldub_code(memaddr + i);

    }

    return 0;

}
