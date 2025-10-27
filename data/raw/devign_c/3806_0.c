static void iter_func(QObject *obj, void *opaque)

{

    QInt *qi;



    fail_unless(opaque == NULL);



    qi = qobject_to_qint(obj);

    fail_unless(qi != NULL);

    fail_unless((qint_get_int(qi) >= 0) && (qint_get_int(qi) <= iter_max));



    iter_called++;

}
