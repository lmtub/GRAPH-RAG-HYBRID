static void type_initialize_interface(TypeImpl *ti, const char *parent)

{

    InterfaceClass *new_iface;

    TypeInfo info = { };

    TypeImpl *iface_impl;



    info.parent = parent;

    info.name = g_strdup_printf("%s::%s", ti->name, info.parent);

    info.abstract = true;



    iface_impl = type_register(&info);

    type_initialize(iface_impl);

    g_free((char *)info.name);



    new_iface = (InterfaceClass *)iface_impl->class;

    new_iface->concrete_class = ti->class;



    ti->class->interfaces = g_slist_append(ti->class->interfaces,

                                           iface_impl->class);

}
