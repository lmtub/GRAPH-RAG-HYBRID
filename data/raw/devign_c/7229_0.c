void qemu_unregister_reset(QEMUResetHandler *func, void *opaque)

{

    QEMUResetEntry *re;



    TAILQ_FOREACH(re, &reset_handlers, entry) {

        if (re->func == func && re->opaque == opaque) {

            TAILQ_REMOVE(&reset_handlers, re, entry);

            qemu_free(re);

            return;

        }

    }

}
