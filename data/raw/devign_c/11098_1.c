MemdevList *qmp_query_memdev(Error **errp)

{

    Object *obj;

    MemdevList *list = NULL;



    obj = object_get_objects_root();

    if (obj == NULL) {

        return NULL;

    }



    if (object_child_foreach(obj, query_memdev, &list) != 0) {

        goto error;

    }



    return list;



error:

    qapi_free_MemdevList(list);

    return NULL;

}
