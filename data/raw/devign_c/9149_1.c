static void load_elf_image(const char *image_name, int image_fd,

                           struct image_info *info, char **pinterp_name,

                           char bprm_buf[BPRM_BUF_SIZE])

{

    struct elfhdr *ehdr = (struct elfhdr *)bprm_buf;

    struct elf_phdr *phdr;

    abi_ulong load_addr, load_bias, loaddr, hiaddr, error;

    int i, retval;

    const char *errmsg;



    /* First of all, some simple consistency checks */

    errmsg = "Invalid ELF image for this architecture";

    if (!elf_check_ident(ehdr)) {

        goto exit_errmsg;

    }

    bswap_ehdr(ehdr);

    if (!elf_check_ehdr(ehdr)) {

        goto exit_errmsg;

    }



    i = ehdr->e_phnum * sizeof(struct elf_phdr);

    if (ehdr->e_phoff + i <= BPRM_BUF_SIZE) {

        phdr = (struct elf_phdr *)(bprm_buf + ehdr->e_phoff);

    } else {

        phdr = (struct elf_phdr *) alloca(i);

        retval = pread(image_fd, phdr, i, ehdr->e_phoff);

        if (retval != i) {

            goto exit_read;

        }

    }

    bswap_phdr(phdr, ehdr->e_phnum);



#ifdef CONFIG_USE_FDPIC

    info->nsegs = 0;

    info->pt_dynamic_addr = 0;

#endif



    /* Find the maximum size of the image and allocate an appropriate

       amount of memory to handle that.  */

    loaddr = -1, hiaddr = 0;

    for (i = 0; i < ehdr->e_phnum; ++i) {

        if (phdr[i].p_type == PT_LOAD) {

            abi_ulong a = phdr[i].p_vaddr;

            if (a < loaddr) {

                loaddr = a;

            }

            a += phdr[i].p_memsz;

            if (a > hiaddr) {

                hiaddr = a;

            }

#ifdef CONFIG_USE_FDPIC

            ++info->nsegs;

#endif

        }

    }



    load_addr = loaddr;

    if (ehdr->e_type == ET_DYN) {

        /* The image indicates that it can be loaded anywhere.  Find a

           location that can hold the memory space required.  If the

           image is pre-linked, LOADDR will be non-zero.  Since we do

           not supply MAP_FIXED here we'll use that address if and

           only if it remains available.  */

        load_addr = target_mmap(loaddr, hiaddr - loaddr, PROT_NONE,

                                MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,

                                -1, 0);

        if (load_addr == -1) {

            goto exit_perror;

        }

    } else if (pinterp_name != NULL) {

        /* This is the main executable.  Make sure that the low

           address does not conflict with MMAP_MIN_ADDR or the

           QEMU application itself.  */

        probe_guest_base(image_name, loaddr, hiaddr);

    }

    load_bias = load_addr - loaddr;



#ifdef CONFIG_USE_FDPIC

    {

        struct elf32_fdpic_loadseg *loadsegs = info->loadsegs =

            g_malloc(sizeof(*loadsegs) * info->nsegs);



        for (i = 0; i < ehdr->e_phnum; ++i) {

            switch (phdr[i].p_type) {

            case PT_DYNAMIC:

                info->pt_dynamic_addr = phdr[i].p_vaddr + load_bias;

                break;

            case PT_LOAD:

                loadsegs->addr = phdr[i].p_vaddr + load_bias;

                loadsegs->p_vaddr = phdr[i].p_vaddr;

                loadsegs->p_memsz = phdr[i].p_memsz;

                ++loadsegs;

                break;

            }

        }

    }

#endif



    info->load_bias = load_bias;

    info->load_addr = load_addr;

    info->entry = ehdr->e_entry + load_bias;

    info->start_code = -1;

    info->end_code = 0;

    info->start_data = -1;

    info->end_data = 0;

    info->brk = 0;




    for (i = 0; i < ehdr->e_phnum; i++) {

        struct elf_phdr *eppnt = phdr + i;

        if (eppnt->p_type == PT_LOAD) {

            abi_ulong vaddr, vaddr_po, vaddr_ps, vaddr_ef, vaddr_em;

            int elf_prot = 0;



            if (eppnt->p_flags & PF_R) elf_prot =  PROT_READ;

            if (eppnt->p_flags & PF_W) elf_prot |= PROT_WRITE;

            if (eppnt->p_flags & PF_X) elf_prot |= PROT_EXEC;



            vaddr = load_bias + eppnt->p_vaddr;

            vaddr_po = TARGET_ELF_PAGEOFFSET(vaddr);

            vaddr_ps = TARGET_ELF_PAGESTART(vaddr);



            error = target_mmap(vaddr_ps, eppnt->p_filesz + vaddr_po,

                                elf_prot, MAP_PRIVATE | MAP_FIXED,

                                image_fd, eppnt->p_offset - vaddr_po);

            if (error == -1) {

                goto exit_perror;

            }



            vaddr_ef = vaddr + eppnt->p_filesz;

            vaddr_em = vaddr + eppnt->p_memsz;



            /* If the load segment requests extra zeros (e.g. bss), map it.  */

            if (vaddr_ef < vaddr_em) {

                zero_bss(vaddr_ef, vaddr_em, elf_prot);

            }



            /* Find the full program boundaries.  */

            if (elf_prot & PROT_EXEC) {

                if (vaddr < info->start_code) {

                    info->start_code = vaddr;

                }

                if (vaddr_ef > info->end_code) {

                    info->end_code = vaddr_ef;

                }

            }

            if (elf_prot & PROT_WRITE) {

                if (vaddr < info->start_data) {

                    info->start_data = vaddr;

                }

                if (vaddr_ef > info->end_data) {

                    info->end_data = vaddr_ef;

                }

                if (vaddr_em > info->brk) {

                    info->brk = vaddr_em;

                }

            }

        } else if (eppnt->p_type == PT_INTERP && pinterp_name) {

            char *interp_name;



            if (*pinterp_name) {

                errmsg = "Multiple PT_INTERP entries";

                goto exit_errmsg;

            }

            interp_name = malloc(eppnt->p_filesz);

            if (!interp_name) {

                goto exit_perror;

            }



            if (eppnt->p_offset + eppnt->p_filesz <= BPRM_BUF_SIZE) {

                memcpy(interp_name, bprm_buf + eppnt->p_offset,

                       eppnt->p_filesz);

            } else {

                retval = pread(image_fd, interp_name, eppnt->p_filesz,

                               eppnt->p_offset);

                if (retval != eppnt->p_filesz) {

                    goto exit_perror;

                }

            }

            if (interp_name[eppnt->p_filesz - 1] != 0) {

                errmsg = "Invalid PT_INTERP entry";

                goto exit_errmsg;

            }

            *pinterp_name = interp_name;

        }

    }



    if (info->end_data == 0) {

        info->start_data = info->end_code;

        info->end_data = info->end_code;

        info->brk = info->end_code;

    }



    if (qemu_log_enabled()) {

        load_symbols(ehdr, image_fd, load_bias);

    }



    close(image_fd);

    return;



 exit_read:

    if (retval >= 0) {

        errmsg = "Incomplete read of file header";

        goto exit_errmsg;

    }

 exit_perror:

    errmsg = strerror(errno);

 exit_errmsg:

    fprintf(stderr, "%s: %s\n", image_name, errmsg);

    exit(-1);

}