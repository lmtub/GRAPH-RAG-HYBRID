START_TEST(unterminated_array_comma)

{

    QObject *obj = qobject_from_json("[32,");

    fail_unless(obj == NULL);

}
