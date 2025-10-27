static int fill_psinfo(struct target_elf_prpsinfo *psinfo, const TaskState *ts)

{

    char *filename, *base_filename;

    unsigned int i, len;



    (void) memset(psinfo, 0, sizeof (*psinfo));



    len = ts->info->arg_end - ts->info->arg_start;

    if (len >= ELF_PRARGSZ)

        len = ELF_PRARGSZ - 1;

    if (copy_from_user(&psinfo->pr_psargs, ts->info->arg_start, len))

        return -EFAULT;

    for (i = 0; i < len; i++)

        if (psinfo->pr_psargs[i] == 0)

            psinfo->pr_psargs[i] = ' ';

    psinfo->pr_psargs[len] = 0;



    psinfo->pr_pid = getpid();

    psinfo->pr_ppid = getppid();

    psinfo->pr_pgrp = getpgrp();

    psinfo->pr_sid = getsid(0);

    psinfo->pr_uid = getuid();

    psinfo->pr_gid = getgid();



    filename = strdup(ts->bprm->filename);

    base_filename = strdup(basename(filename));

    (void) strncpy(psinfo->pr_fname, base_filename,

                   sizeof(psinfo->pr_fname));

    free(base_filename);

    free(filename);



#ifdef BSWAP_NEEDED

    bswap_psinfo(psinfo);

#endif

    return (0);

}
