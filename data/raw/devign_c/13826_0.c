static void qapi_clone_type_str(Visitor *v, const char *name, char **obj,

                                 Error **errp)

{

    QapiCloneVisitor *qcv = to_qcv(v);



    assert(qcv->depth);

    /*

     * Pointer was already cloned by g_memdup; create fresh copy.

     * Note that as long as qmp-output-visitor accepts NULL instead of

     * "", then we must do likewise. However, we want to obey the

     * input visitor semantics of never producing NULL when the empty

     * string is intended.

     */

    *obj = g_strdup(*obj ?: "");

}
