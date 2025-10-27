void qemu_del_vm_change_state_handler(VMChangeStateEntry *e)

{

    LIST_REMOVE (e, entries);

    qemu_free (e);

}
