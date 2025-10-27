START_TEST(unterminated_array)

{

    QObject *obj = qobject_from_json("[32");

    fail_unless(obj == NULL);

}
