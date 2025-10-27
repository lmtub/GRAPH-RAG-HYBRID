static void evalPrimary(Parser *p){

    double d, d2=NAN;

    char *next= p->s;

    int i;



    /* number */

    d= strtod(p->s, &next);

    if(next != p->s){

        push(p, d);

        p->s= next;

        return;

    }

    

    /* named constants */

    for(i=0; p->const_name[i]; i++){

        if(strmatch(p->s, p->const_name[i])){

            push(p, p->const_value[i]);

            p->s+= strlen(p->const_name[i]);

            return;

        }

    }

    

    p->s= strchr(p->s, '(');

    if(p->s==NULL){

        av_log(NULL, AV_LOG_ERROR, "Parser: missing ( in \"%s\"\n", next);

        return;

    }

    p->s++; // "("

    evalExpression(p);

    d= pop(p);

    p->s++; // ")" or ","

    if(p->s[-1]== ','){

        evalExpression(p);

        d2= pop(p);

        p->s++; // ")"

    }

    

         if( strmatch(next, "sinh"  ) ) d= sinh(d);

    else if( strmatch(next, "cosh"  ) ) d= cosh(d);

    else if( strmatch(next, "tanh"  ) ) d= tanh(d);

    else if( strmatch(next, "sin"   ) ) d= sin(d);

    else if( strmatch(next, "cos"   ) ) d= cos(d);

    else if( strmatch(next, "tan"   ) ) d= tan(d);

    else if( strmatch(next, "exp"   ) ) d= exp(d);

    else if( strmatch(next, "log"   ) ) d= log(d);

    else if( strmatch(next, "squish") ) d= 1/(1+exp(4*d));

    else if( strmatch(next, "gauss" ) ) d= exp(-d*d/2)/sqrt(2*M_PI);

    else if( strmatch(next, "abs"   ) ) d= fabs(d);

    else if( strmatch(next, "max"   ) ) d= d > d2 ? d : d2;

    else if( strmatch(next, "min"   ) ) d= d < d2 ? d : d2;

    else if( strmatch(next, "gt"    ) ) d= d > d2 ? 1.0 : 0.0;

    else if( strmatch(next, "lt"    ) ) d= d > d2 ? 0.0 : 1.0;

    else if( strmatch(next, "eq"    ) ) d= d == d2 ? 1.0 : 0.0;

//    else if( strmatch(next, "l1"    ) ) d= 1 + d2*(d - 1);

//    else if( strmatch(next, "sq01"  ) ) d= (d >= 0.0 && d <=1.0) ? 1.0 : 0.0;

    else{

        int error=1;

        for(i=0; p->func1_name && p->func1_name[i]; i++){

            if(strmatch(next, p->func1_name[i])){

                d= p->func1[i](p->opaque, d);

                error=0;

                break;

            }

        }



        for(i=0; p->func2_name && p->func2_name[i]; i++){

            if(strmatch(next, p->func2_name[i])){

                d= p->func2[i](p->opaque, d, d2);

                error=0;

                break;

            }

        }



        if(error){

            av_log(NULL, AV_LOG_ERROR, "Parser: unknown function in \"%s\"\n", next);

            return;

        }

    }

    

    if(p->s[-1]!= ')'){

        av_log(NULL, AV_LOG_ERROR, "Parser: missing ) in \"%s\"\n", next);

        return;

    }

    push(p, d);

}      
