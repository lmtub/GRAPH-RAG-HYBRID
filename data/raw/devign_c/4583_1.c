ObjectClass *object_class_dynamic_cast_assert(ObjectClass *class,

                                              const char *typename,

                                              const char *file, int line,

                                              const char *func)

{

    ObjectClass *ret = object_class_dynamic_cast(class, typename);



    if (!ret && class) {

        fprintf(stderr, "%s:%d:%s: Object %p is not an instance of type %s\n",

                file, line, func, class, typename);

        abort();

    }



    return ret;

}
