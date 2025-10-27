void json_end_object(QJSON *json)

{

    qstring_append(json->str, " }");

    json->omit_comma = false;

}
