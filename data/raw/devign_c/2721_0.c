START_TEST(unterminated_dict_comma)

{

    QObject *obj = qobject_from_json("{'abc':32,");

    fail_unless(obj == NULL);

}
