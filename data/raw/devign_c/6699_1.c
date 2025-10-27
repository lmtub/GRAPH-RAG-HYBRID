void bdrv_image_info_specific_dump(fprintf_function func_fprintf, void *f,

                                   ImageInfoSpecific *info_spec)

{

    QObject *obj, *data;

    Visitor *v = qmp_output_visitor_new(&obj);



    visit_type_ImageInfoSpecific(v, NULL, &info_spec, &error_abort);

    visit_complete(v, &obj);

    assert(qobject_type(obj) == QTYPE_QDICT);

    data = qdict_get(qobject_to_qdict(obj), "data");

    dump_qobject(func_fprintf, f, 1, data);


    visit_free(v);

}