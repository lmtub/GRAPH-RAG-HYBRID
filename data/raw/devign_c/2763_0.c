void qmp_screendump(const char *filename, Error **errp)

{

    QemuConsole *previous_active_console;

    bool cswitch;



    previous_active_console = active_console;

    cswitch = previous_active_console && previous_active_console->index != 0;



    /* There is currently no way of specifying which screen we want to dump,

       so always dump the first one.  */

    if (cswitch) {

        console_select(0);

    }

    if (consoles[0] && consoles[0]->hw_screen_dump) {

        consoles[0]->hw_screen_dump(consoles[0]->hw, filename, cswitch, errp);

    } else {

        error_setg(errp, "device doesn't support screendump");

    }



    if (cswitch) {

        console_select(previous_active_console->index);

    }

}
