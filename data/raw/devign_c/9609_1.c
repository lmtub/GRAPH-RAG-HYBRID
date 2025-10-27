int loader_exec(const char * filename, char ** argv, char ** envp,

             struct target_pt_regs * regs, struct image_info *infop)

{

    struct linux_binprm bprm;

    int retval;

    int i;



    bprm.p = TARGET_PAGE_SIZE*MAX_ARG_PAGES-sizeof(unsigned int);

    for (i=0 ; i<MAX_ARG_PAGES ; i++)       /* clear page-table */

            bprm.page[i] = NULL;

    retval = open(filename, O_RDONLY);

    if (retval < 0)

        return retval;

    bprm.fd = retval;

    bprm.filename = (char *)filename;

    bprm.argc = count(argv);

    bprm.argv = argv;

    bprm.envc = count(envp);

    bprm.envp = envp;



    retval = prepare_binprm(&bprm);



    if(retval>=0) {

        if (bprm.buf[0] == 0x7f

                && bprm.buf[1] == 'E'

                && bprm.buf[2] == 'L'

                && bprm.buf[3] == 'F') {

            retval = load_elf_binary(&bprm,regs,infop);

        } else {

            fprintf(stderr, "Unknown binary format\n");

            return -1;

        }

    }



    if(retval>=0) {

        /* success.  Initialize important registers */

        do_init_thread(regs, infop);

        return retval;

    }



    /* Something went wrong, return the inode and free the argument pages*/

    for (i=0 ; i<MAX_ARG_PAGES ; i++) {

        free(bprm.page[i]);

    }

    return(retval);

}
