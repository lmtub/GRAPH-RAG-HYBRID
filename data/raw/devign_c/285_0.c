static void qjson_finalizefn(Object *obj)

{

    QJSON *json = QJSON(obj);



    qobject_decref(QOBJECT(json->str));

}
