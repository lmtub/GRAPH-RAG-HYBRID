bool qemu_savevm_state_blocked(Monitor *mon)

{

    SaveStateEntry *se;



    QTAILQ_FOREACH(se, &savevm_handlers, entry) {

        if (se->no_migrate) {

            monitor_printf(mon, "state blocked by non-migratable device '%s'\n",

                           se->idstr);

            return true;

        }

    }

    return false;

}
