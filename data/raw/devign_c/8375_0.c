START_TEST(unterminated_escape)

{

    QObject *obj = qobject_from_json("\"abc\\\"");

    fail_unless(obj == NULL);

}
