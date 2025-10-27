static void unterminated_literal(void)

{

    QObject *obj = qobject_from_json("nul", NULL);

    g_assert(obj == NULL);

}
