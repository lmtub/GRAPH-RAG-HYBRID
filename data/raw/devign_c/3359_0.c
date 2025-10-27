int qemu_add_child_watch(pid_t pid)

{

    ChildProcessRecord *rec;



    if (!sigchld_bh) {

        qemu_init_child_watch();

    }



    QLIST_FOREACH(rec, &child_watches, next) {

        if (rec->pid == pid) {

            return 1;

        }

    }

    rec = g_malloc0(sizeof(ChildProcessRecord));

    rec->pid = pid;

    QLIST_INSERT_HEAD(&child_watches, rec, next);

    return 0;

}
