static int load_flat_file(struct linux_binprm * bprm,

		struct lib_info *libinfo, int id, abi_ulong *extra_stack)

{

    struct flat_hdr * hdr;

    abi_ulong textpos = 0, datapos = 0;

    abi_long result;

    abi_ulong realdatastart = 0;

    abi_ulong text_len, data_len, bss_len, stack_len, flags;

    abi_ulong memp = 0; /* for finding the brk area */

    abi_ulong extra;

    abi_ulong reloc = 0, rp;

    int i, rev, relocs = 0;

    abi_ulong fpos;

    abi_ulong start_code, end_code;

    abi_ulong indx_len;



    hdr = ((struct flat_hdr *) bprm->buf);		/* exec-header */



    text_len  = ntohl(hdr->data_start);

    data_len  = ntohl(hdr->data_end) - ntohl(hdr->data_start);

    bss_len   = ntohl(hdr->bss_end) - ntohl(hdr->data_end);

    stack_len = ntohl(hdr->stack_size);

    if (extra_stack) {

        stack_len += *extra_stack;

        *extra_stack = stack_len;

    }

    relocs    = ntohl(hdr->reloc_count);

    flags     = ntohl(hdr->flags);

    rev       = ntohl(hdr->rev);



    DBG_FLT("BINFMT_FLAT: Loading file: %s\n", bprm->filename);



    if (rev != FLAT_VERSION && rev != OLD_FLAT_VERSION) {

        fprintf(stderr, "BINFMT_FLAT: bad magic/rev (0x%x, need 0x%x)\n",

                rev, (int) FLAT_VERSION);

        return -ENOEXEC;

    }



    /* Don't allow old format executables to use shared libraries */

    if (rev == OLD_FLAT_VERSION && id != 0) {

        fprintf(stderr, "BINFMT_FLAT: shared libraries are not available\n");

        return -ENOEXEC;

    }



    /*

     * fix up the flags for the older format,  there were all kinds

     * of endian hacks,  this only works for the simple cases

     */

    if (rev == OLD_FLAT_VERSION && flat_old_ram_flag(flags))

        flags = FLAT_FLAG_RAM;



#ifndef CONFIG_BINFMT_ZFLAT

    if (flags & (FLAT_FLAG_GZIP|FLAT_FLAG_GZDATA)) {

        fprintf(stderr, "Support for ZFLAT executables is not enabled\n");

        return -ENOEXEC;

    }

#endif



    /*

     * calculate the extra space we need to map in

     */

    extra = relocs * sizeof(abi_ulong);

    if (extra < bss_len + stack_len)

        extra = bss_len + stack_len;



    /* Add space for library base pointers.  Make sure this does not

       misalign the  doesn't misalign the data segment.  */

    indx_len = MAX_SHARED_LIBS * sizeof(abi_ulong);

    indx_len = (indx_len + 15) & ~(abi_ulong)15;



    /*

     * there are a couple of cases here,  the separate code/data

     * case,  and then the fully copied to RAM case which lumps

     * it all together.

     */

    if ((flags & (FLAT_FLAG_RAM|FLAT_FLAG_GZIP)) == 0) {

        /*

         * this should give us a ROM ptr,  but if it doesn't we don't

         * really care

         */

        DBG_FLT("BINFMT_FLAT: ROM mapping of file (we hope)\n");



        textpos = target_mmap(0, text_len, PROT_READ|PROT_EXEC,

                              MAP_PRIVATE, bprm->fd, 0);

        if (textpos == -1) {

            fprintf(stderr, "Unable to mmap process text\n");

            return -1;

        }



        realdatastart = target_mmap(0, data_len + extra + indx_len,

                                    PROT_READ|PROT_WRITE|PROT_EXEC,

                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);



        if (realdatastart == -1) {

            fprintf(stderr, "Unable to allocate RAM for process data\n");

            return realdatastart;

        }

        datapos = realdatastart + indx_len;



        DBG_FLT("BINFMT_FLAT: Allocated data+bss+stack (%d bytes): %x\n",

                        (int)(data_len + bss_len + stack_len), (int)datapos);



        fpos = ntohl(hdr->data_start);

#ifdef CONFIG_BINFMT_ZFLAT

        if (flags & FLAT_FLAG_GZDATA) {

            result = decompress_exec(bprm, fpos, (char *) datapos,

                                     data_len + (relocs * sizeof(abi_ulong)))

        } else

#endif

        {

            result = target_pread(bprm->fd, datapos,

                                  data_len + (relocs * sizeof(abi_ulong)),

                                  fpos);

        }

        if (result < 0) {

            fprintf(stderr, "Unable to read data+bss\n");

            return result;

        }



        reloc = datapos + (ntohl(hdr->reloc_start) - text_len);

        memp = realdatastart;



    } else {



        textpos = target_mmap(0, text_len + data_len + extra + indx_len,

                              PROT_READ | PROT_EXEC | PROT_WRITE,

                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (textpos == -1 ) {

            fprintf(stderr, "Unable to allocate RAM for process text/data\n");

            return -1;

        }



        realdatastart = textpos + ntohl(hdr->data_start);

        datapos = realdatastart + indx_len;

        reloc = (textpos + ntohl(hdr->reloc_start) + indx_len);

        memp = textpos;



#ifdef CONFIG_BINFMT_ZFLAT

#error code needs checking

        /*

         * load it all in and treat it like a RAM load from now on

         */

        if (flags & FLAT_FLAG_GZIP) {

                result = decompress_exec(bprm, sizeof (struct flat_hdr),

                                 (((char *) textpos) + sizeof (struct flat_hdr)),

                                 (text_len + data_len + (relocs * sizeof(unsigned long))

                                          - sizeof (struct flat_hdr)),

                                 0);

                memmove((void *) datapos, (void *) realdatastart,

                                data_len + (relocs * sizeof(unsigned long)));

        } else if (flags & FLAT_FLAG_GZDATA) {

                fpos = 0;

                result = bprm->file->f_op->read(bprm->file,

                                (char *) textpos, text_len, &fpos);

                if (result < (unsigned long) -4096)

                        result = decompress_exec(bprm, text_len, (char *) datapos,

                                         data_len + (relocs * sizeof(unsigned long)), 0);

        }

        else

#endif

        {

            result = target_pread(bprm->fd, textpos,

                                  text_len, 0);

            if (result >= 0) {

                result = target_pread(bprm->fd, datapos,

                    data_len + (relocs * sizeof(abi_ulong)),

                    ntohl(hdr->data_start));

            }

        }

        if (result < 0) {

            fprintf(stderr, "Unable to read code+data+bss\n");

            return result;

        }

    }



    DBG_FLT("Mapping is 0x%x, Entry point is 0x%x, data_start is 0x%x\n",

            (int)textpos, 0x00ffffff&ntohl(hdr->entry),

            ntohl(hdr->data_start));



    /* The main program needs a little extra setup in the task structure */

    start_code = textpos + sizeof (struct flat_hdr);

    end_code = textpos + text_len;



    DBG_FLT("%s %s: TEXT=%x-%x DATA=%x-%x BSS=%x-%x\n",

            id ? "Lib" : "Load", bprm->filename,

            (int) start_code, (int) end_code,

            (int) datapos,

            (int) (datapos + data_len),

            (int) (datapos + data_len),

            (int) (((datapos + data_len + bss_len) + 3) & ~3));



    text_len -= sizeof(struct flat_hdr); /* the real code len */



    /* Store the current module values into the global library structure */

    libinfo[id].start_code = start_code;

    libinfo[id].start_data = datapos;

    libinfo[id].end_data = datapos + data_len;

    libinfo[id].start_brk = datapos + data_len + bss_len;

    libinfo[id].text_len = text_len;

    libinfo[id].loaded = 1;

    libinfo[id].entry = (0x00ffffff & ntohl(hdr->entry)) + textpos;

    libinfo[id].build_date = ntohl(hdr->build_date);



    /*

     * We just load the allocations into some temporary memory to

     * help simplify all this mumbo jumbo

     *

     * We've got two different sections of relocation entries.

     * The first is the GOT which resides at the begining of the data segment

     * and is terminated with a -1.  This one can be relocated in place.

     * The second is the extra relocation entries tacked after the image's

     * data segment. These require a little more processing as the entry is

     * really an offset into the image which contains an offset into the

     * image.

     */

    if (flags & FLAT_FLAG_GOTPIC) {

        rp = datapos;

        while (1) {

            abi_ulong addr;

            if (get_user_ual(addr, rp))

                return -EFAULT;

            if (addr == -1)

                break;

            if (addr) {

                addr = calc_reloc(addr, libinfo, id, 0);

                if (addr == RELOC_FAILED)

                    return -ENOEXEC;

                if (put_user_ual(addr, rp))

                    return -EFAULT;

            }

            rp += sizeof(abi_ulong);

        }

    }



    /*

     * Now run through the relocation entries.

     * We've got to be careful here as C++ produces relocatable zero

     * entries in the constructor and destructor tables which are then

     * tested for being not zero (which will always occur unless we're

     * based from address zero).  This causes an endless loop as __start

     * is at zero.  The solution used is to not relocate zero addresses.

     * This has the negative side effect of not allowing a global data

     * reference to be statically initialised to _stext (I've moved

     * __start to address 4 so that is okay).

     */

    if (rev > OLD_FLAT_VERSION) {

        for (i = 0; i < relocs; i++) {

            abi_ulong addr, relval;



            /* Get the address of the pointer to be

               relocated (of course, the address has to be

               relocated first).  */

            if (get_user_ual(relval, reloc + i * sizeof(abi_ulong)))

                return -EFAULT;

            addr = flat_get_relocate_addr(relval);

            rp = calc_reloc(addr, libinfo, id, 1);

            if (rp == RELOC_FAILED)

                return -ENOEXEC;



            /* Get the pointer's value.  */

            if (get_user_ual(addr, rp))

                return -EFAULT;

            if (addr != 0) {

                /*

                 * Do the relocation.  PIC relocs in the data section are

                 * already in target order

                 */



#ifndef TARGET_WORDS_BIGENDIAN

                if ((flags & FLAT_FLAG_GOTPIC) == 0)

                    addr = bswap32(addr);

#endif

                addr = calc_reloc(addr, libinfo, id, 0);

                if (addr == RELOC_FAILED)

                    return -ENOEXEC;



                /* Write back the relocated pointer.  */

                if (put_user_ual(addr, rp))

                    return -EFAULT;

            }

        }

    } else {

        for (i = 0; i < relocs; i++) {

            abi_ulong relval;

            if (get_user_ual(relval, reloc + i * sizeof(abi_ulong)))

                return -EFAULT;

            old_reloc(&libinfo[0], relval);

        }

    }



    /* zero the BSS.  */

    memset((void *)((unsigned long)datapos + data_len), 0, bss_len);



    return 0;

}
