static void do_loadvm(Monitor *mon, const QDict *qdict)

{

    int saved_vm_running  = vm_running;

    const char *name = qdict_get_str(qdict, "name");



    vm_stop(0);



    if (load_vmstate(name) >= 0 && saved_vm_running)

        vm_start();

}
