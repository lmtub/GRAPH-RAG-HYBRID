static QDict *build_qmp_error_dict(const QError *err)

{

    QObject *obj;



    obj = qobject_from_jsonf("{ 'error': { 'class': %s, 'desc': %p } }",

                             ErrorClass_lookup[err->err_class],

                             qerror_human(err));



    return qobject_to_qdict(obj);

}
