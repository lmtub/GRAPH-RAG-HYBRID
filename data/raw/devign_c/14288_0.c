void json_end_array(QJSON *json)

{

    qstring_append(json->str, " ]");

    json->omit_comma = false;

}
