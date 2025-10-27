static void type_initialize(TypeImpl *ti)

{

    TypeImpl *parent;



    if (ti->class) {

        return;

    }



    ti->class_size = type_class_get_size(ti);

    ti->instance_size = type_object_get_size(ti);



    ti->class = g_malloc0(ti->class_size);



    parent = type_get_parent(ti);

    if (parent) {

        type_initialize(parent);

        GSList *e;

        int i;



        g_assert(parent->class_size <= ti->class_size);

        memcpy(ti->class, parent->class, parent->class_size);

        ti->class->interfaces = NULL;



        for (e = parent->class->interfaces; e; e = e->next) {

            ObjectClass *iface = e->data;

            type_initialize_interface(ti, object_class_get_name(iface));

        }



        for (i = 0; i < ti->num_interfaces; i++) {

            TypeImpl *t = type_get_by_name(ti->interfaces[i].typename);

            for (e = ti->class->interfaces; e; e = e->next) {

                TypeImpl *target_type = OBJECT_CLASS(e->data)->type;



                if (type_is_ancestor(target_type, t)) {

                    break;

                }

            }



            if (e) {

                continue;

            }



            type_initialize_interface(ti, ti->interfaces[i].typename);

        }

    }



    ti->class->type = ti;



    while (parent) {

        if (parent->class_base_init) {

            parent->class_base_init(ti->class, ti->class_data);

        }

        parent = type_get_parent(parent);

    }



    if (ti->class_init) {

        ti->class_init(ti->class, ti->class_data);

    }





}
