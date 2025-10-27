static void qjson_initfn(Object *obj)

{

    QJSON *json = QJSON(obj);



    json->str = qstring_from_str("{ ");

    json->omit_comma = true;

}
