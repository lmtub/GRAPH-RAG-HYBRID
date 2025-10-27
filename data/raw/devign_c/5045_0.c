static void sigchld_bh_handler(void *opaque)

{

    ChildProcessRecord *rec, *next;



    QLIST_FOREACH_SAFE(rec, &child_watches, next, next) {

        if (waitpid(rec->pid, NULL, WNOHANG) == rec->pid) {

            QLIST_REMOVE(rec, next);

            g_free(rec);

        }

    }

}
