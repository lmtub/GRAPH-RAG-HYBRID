static QemuConsole *new_console(DisplayState *ds, console_type_t console_type)

{

    Error *local_err = NULL;

    Object *obj;

    QemuConsole *s;

    int i;



    if (nb_consoles >= MAX_CONSOLES)

        return NULL;



    obj = object_new(TYPE_QEMU_CONSOLE);

    s = QEMU_CONSOLE(obj);

    object_property_add_link(obj, "device", TYPE_DEVICE,

                             (Object **)&s->device,

                             object_property_allow_set_link,

                             OBJ_PROP_LINK_UNREF_ON_RELEASE,

                             &local_err);

    object_property_add_uint32_ptr(obj, "head",

                                   &s->head, &local_err);



    if (!active_console || ((active_console->console_type != GRAPHIC_CONSOLE) &&

        (console_type == GRAPHIC_CONSOLE))) {

        active_console = s;

    }

    s->ds = ds;

    s->console_type = console_type;

    if (console_type != GRAPHIC_CONSOLE) {

        s->index = nb_consoles;

        consoles[nb_consoles++] = s;

    } else {

        /* HACK: Put graphical consoles before text consoles.  */

        for (i = nb_consoles; i > 0; i--) {

            if (consoles[i - 1]->console_type == GRAPHIC_CONSOLE)

                break;

            consoles[i] = consoles[i - 1];

            consoles[i]->index = i;

        }

        s->index = i;

        consoles[i] = s;

        nb_consoles++;

    }

    return s;

}
