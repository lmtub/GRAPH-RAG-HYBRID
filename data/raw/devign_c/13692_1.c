static void empty_input(void)

{

    const char *empty = "";



    QObject *obj = qobject_from_json(empty, NULL);

    g_assert(obj == NULL);

}
