static QmpInputVisitor *to_qiv(Visitor *v)

{

    return container_of(v, QmpInputVisitor, visitor);

}
