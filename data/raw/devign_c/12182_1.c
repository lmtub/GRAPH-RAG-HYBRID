const char *object_get_typename(Object *obj)

{

    return obj->class->type->name;

}
