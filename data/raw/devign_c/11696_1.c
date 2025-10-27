void machine_register_compat_props(MachineState *machine)

{

    MachineClass *mc = MACHINE_GET_CLASS(machine);

    int i;

    GlobalProperty *p;



    if (!mc->compat_props) {

        return;

    }



    for (i = 0; i < mc->compat_props->len; i++) {

        p = g_array_index(mc->compat_props, GlobalProperty *, i);



        qdev_prop_register_global(p);

    }

}