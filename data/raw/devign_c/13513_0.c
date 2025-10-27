void qemu_register_reset(QEMUResetHandler *func, void *opaque)

{

    QEMUResetEntry *re = qemu_mallocz(sizeof(QEMUResetEntry));



    re->func = func;

    re->opaque = opaque;

    TAILQ_INSERT_TAIL(&reset_handlers, re, entry);

}
