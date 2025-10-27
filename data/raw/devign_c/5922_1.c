Object *object_dynamic_cast(Object *obj, const char *typename)

{

    if (object_class_dynamic_cast(object_get_class(obj), typename)) {

        return obj;

    }



    return NULL;

}
