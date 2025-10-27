Object *object_dynamic_cast(Object *obj, const char *typename)

{

    GSList *i;



    /* Check if typename is a direct ancestor */

    if (object_is_type(obj, typename)) {

        return obj;

    }



    /* Check if obj has an interface of typename */

    for (i = obj->interfaces; i; i = i->next) {

        Interface *iface = i->data;



        if (object_is_type(OBJECT(iface), typename)) {

            return OBJECT(iface);

        }

    }



    /* Check if obj is an interface and its containing object is a direct

     * ancestor of typename */

    if (object_is_type(obj, TYPE_INTERFACE)) {

        Interface *iface = INTERFACE(obj);



        if (object_is_type(iface->obj, typename)) {

            return iface->obj;

        }

    }



    return NULL;

}
