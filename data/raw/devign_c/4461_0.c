VMChangeStateEntry *qemu_add_vm_change_state_handler(VMChangeStateHandler *cb,

                                                     void *opaque)

{

    VMChangeStateEntry *e;



    e = qemu_mallocz(sizeof (*e));



    e->cb = cb;

    e->opaque = opaque;

    LIST_INSERT_HEAD(&vm_change_state_head, e, entries);

    return e;

}
