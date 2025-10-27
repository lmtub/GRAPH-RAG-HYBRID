static void unterminated_string(void)

{

    QObject *obj = qobject_from_json("\"abc", NULL);

    g_assert(obj == NULL);

}
