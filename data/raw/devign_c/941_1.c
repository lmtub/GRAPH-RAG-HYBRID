static void nvme_get_bootindex(Object *obj, Visitor *v, void *opaque,

                                  const char *name, Error **errp)

{

    NvmeCtrl *s = NVME(obj);



    visit_type_int32(v, &s->conf.bootindex, name, errp);

}
