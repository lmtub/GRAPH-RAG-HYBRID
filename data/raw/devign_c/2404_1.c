static void unterminated_array(void)

{

    QObject *obj = qobject_from_json("[32", NULL);

    g_assert(obj == NULL);

}
