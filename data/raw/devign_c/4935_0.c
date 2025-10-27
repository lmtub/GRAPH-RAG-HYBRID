int qemu_register_machine(QEMUMachine *m)

{

    char *name = g_strconcat(m->name, TYPE_MACHINE_SUFFIX, NULL);

    TypeInfo ti = {

        .name       = name,

        .parent     = TYPE_MACHINE,

        .class_init = machine_class_init,

        .class_data = (void *)m,

    };



    type_register(&ti);

    g_free(name);



    return 0;

}
