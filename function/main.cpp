//
//  main.cpp
//  function
//
//  Created by Nozomu Ishiguro on 2015/06/01.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"
#include <algorithm>

string partialDiff(string formula, string variable){
    cout<<"initial: "<<formula<<endl;
    string diff_string;
    
    size_t index=0;
    size_t index_plus=0, index_minus=0;
    size_t index_multi=0;
    size_t index_devide=0;
    size_t index_bla=0;
    size_t index_ket=0;
    size_t blaketFloor=0;
    size_t index_bla_sub=0;
    size_t index_ket_sub=0;
    
    
    do {
        index_bla=formula.find("(",index);
        index_ket=formula.find(")",index);
        index_plus=formula.find("+",index);
        index_minus=formula.find("-",index);
        
        if (blaketFloor==0) {
            if (index_bla==string::npos) {
                break;
            } else if (index_plus<index_bla) {
                break;
            } else if (index_minus<index_bla) {
                break;
            } else{
                blaketFloor++;
                index = index_bla+1;
                index_bla_sub = index_bla;
            }
        } else {
            if (index_bla < index_ket) {
                blaketFloor++;
                index = index_bla+1;
            } else {
                blaketFloor--;
                index = index_ket+1;
                index_ket_sub = index_ket;
            }
        }
    } while (index<formula.length());
    
    index_bla=formula.find("(",index);
    index_ket=formula.find(")",index);
    index_plus=formula.find("+",index);
    index_minus=formula.find("-",index);
    
    string blaket=formula;
    blaket.erase(index_ket_sub, formula.length());
    blaket.erase(0,index_bla_sub+1);
    
    //cout<<"blaket: "<<blaket<<endl;
    //中身に項が1つのみ
    if (index_plus==string::npos & index_minus==string::npos) {
        //中身が"a" (変数x無し)
        //cout<<formula.find(variable,0);
        if (formula.find(variable)==string::npos) {
            diff_string="";
        }
        //中身がg(x)*(...)^(...)を含む
        else if(formula.find("^("+blaket+")")!=string::npos){
            index = formula.find("^("+blaket+")");
            do {
                index_bla=formula.rfind("(",index);
                index_ket=formula.rfind(")",index);
                index_multi=formula.rfind("*",index);
                index_devide=formula.rfind("/",index);
                
                if (blaketFloor==0) {
                    if (index_multi==string::npos){
                        index_multi=0;
                    }
                    if (index_devide==string::npos){
                        index_devide=0;
                    }
                    if (index_ket==string::npos){
                        index_ket=0;
                    }
                    
                    if (max({index_multi,index_devide})>index_ket) {
                        index=max({index_multi,index_devide});
                        break;
                    } else if(index_ket==string::npos){
                        index=0;
                        break;
                    }else if(index_ket!=string::npos){
                        blaketFloor++;
                        index = index_ket-1;
                        index_bla_sub = index_ket;
                    }
                } else {
                    if (index_bla < index_ket) {
                        blaketFloor--;
                        index = index_bla-1;
                        index_bla_sub = index_bla;
                    } else {
                        blaketFloor++;
                        index = index_ket-1;
                        
                    }
                }
            } while (index>0);
            index_bla=formula.rfind("(",index);
            index_ket=formula.rfind(")",index);
            index_multi=formula.rfind("*",index);
            index_devide=formula.rfind("/",index);
            
            size_t index_end=formula.find("^("+blaket+")")+("^("+blaket+")").length();
            
            string term1 = formula;
            term1 = term1.replace(index,index_end-1,"");
            string term2 = formula;
            term2.erase(index_end);
            term2.erase(0,index);
            string opt=term2;
            term2.erase(0,1);
            opt.erase(1);
            string base=term2;
            base.erase(base.find("^("+blaket+")"));
            base.erase(0,index-1);
            string base1=base;
            if (base1.find("(")==0&base1.find(")")==base1.length()-1) {
                base1.erase(0,1);
                base1.erase(base1.length()-1,1);
            }
            cout<<"term1: "<<term1<<endl;
            cout<<"term2: "<<term2<<endl;
            cout<<"g(x):"<< base<<endl;
            cout<<"h(x):"<< blaket<<endl;
            cout<<"operator: "<<opt<<endl;
            
            string diff_string1=partialDiff(term1,"x");
            if (diff_string1=="1") {
                diff_string1=term2;
            } else if (diff_string1!="") {
                diff_string1+="*"+term2;
            }
            
            string diff_string2,diff_string3,diff_string4;
            diff_string3=partialDiff(base1,"x");
            diff_string4=partialDiff(blaket,"x");
            if (diff_string3==""&diff_string4=="") {
                diff_string2="";
            }else if(diff_string3==""){
                diff_string2=base+"^("+blaket+")"+opt+diff_string4+opt+"ln("+base1+")";
            }else if(diff_string4==""){
                diff_string2=base+"^("+blaket+"-1)"+opt+"("+blaket+")"+opt+base;
            }else{
                diff_string2="("+base+"^("+blaket+")"+opt+diff_string4+opt+"ln("+base1+")+";
                diff_string2+=base+"^("+blaket+"-1)"+opt+"("+blaket+")"+opt+base+")";
            }
            
            if(diff_string1.length()==0 & diff_string2.length()==0){
                diff_string="";
            } else if (diff_string1.length()==0){
                diff_string = term1+opt+diff_string2;
            }else{
                diff_string = diff_string1+opt+term2+"+"+term1+opt+diff_string2;
            }
            //cout<<diff_string2<<endl;
        }
        //中身がg(x)*(...)を含む
        else if(formula.find("*("+blaket+")")!=string::npos){
            size_t loc = formula.find("*("+blaket+")");
            string term1 = formula.replace(loc,blaket.length()+3,"");
            string term2 = "("+blaket+")";
            
            string diff_string1=partialDiff(term1,"x");
            if (diff_string1=="1") {
                diff_string1=term2;
            } else if (diff_string1!="") {
                diff_string1+="*"+term2;
            }
            string diff_string2=partialDiff(blaket,"x");
            index=0;
            do {
                index_bla=diff_string2.find("(",index);
                index_ket=diff_string2.find(")",index);
                index_plus=diff_string2.find("+",index);
                index_minus=diff_string2.find("-",index);
                
                if (blaketFloor==0) {
                    if (index_plus!=string::npos | index_minus!=string::npos) {
                        diff_string2="("+diff_string2+")";
                        break;
                    } else if(index_bla==string::npos) {
                        break;
                    }else{
                        blaketFloor++;
                        index = index_bla+1;
                        index_bla_sub = index_bla;
                    }
                } else {
                    if (index_bla < index_ket) {
                        blaketFloor++;
                        index = index_bla+1;
                    } else {
                        blaketFloor--;
                        index = index_ket+1;
                        index_ket_sub = index_ket;
                    }
                }
            } while (index<formula.length());
            if (diff_string2=="1") {
                diff_string2=term1;
            } else if (diff_string2!="") {
                diff_string2=term1+"*"+diff_string2;
            }
            
            if(diff_string1.length()==0 & diff_string2.length()==0){
                diff_string="";
            } else if (diff_string1.length()==0){
                diff_string = diff_string2;
            } else if (diff_string2.length()==0){
                diff_string = diff_string1;
            }else{
                diff_string = diff_string1+"+"+diff_string2;
            }
            
        }
        //中身がg(x)/(...)を含む
        else if(formula.find("/("+blaket+")")!=string::npos){
            size_t loc = formula.find("/("+blaket+")");
            string term1 = formula.replace(loc,blaket.length()+3,"");
            string term2 = "("+blaket+")";
            string blaket1=term1;
            if (blaket1.find("(")==0&blaket1.find(")")==blaket1.length()-1) {
                blaket1.erase(0,1);
                blaket1.erase(blaket1.length()-1,1);
            }
            
            string diff_string1=partialDiff(blaket1,"x");
            if (diff_string1=="1") {
                diff_string1=term2;
            } else if (diff_string1!="") {
                diff_string1+="/"+term2;
            }
            string diff_string2=partialDiff(blaket,"x");
            index=0;
            do {
                index_bla=diff_string2.find("(",index);
                index_ket=diff_string2.find(")",index);
                index_plus=diff_string2.find("+",index);
                index_minus=diff_string2.find("-",index);
                
                if (blaketFloor==0) {
                    if (index_plus!=string::npos | index_minus!=string::npos) {
                        diff_string2="("+diff_string2+")";
                        break;
                    } else if(index_bla==string::npos) {
                        break;
                    }else{
                        blaketFloor++;
                        index = index_bla+1;
                        index_bla_sub = index_bla;
                    }
                } else {
                    if (index_bla < index_ket) {
                        blaketFloor++;
                        index = index_bla+1;
                    } else {
                        blaketFloor--;
                        index = index_ket+1;
                        index_ket_sub = index_ket;
                    }
                }
            } while (index<formula.length());
            if (diff_string2=="1") {
                diff_string2=term1;
            } else if (diff_string2!="") {
                diff_string2=term1+"/"+term2+"^2*"+diff_string2;
            }
            
            if(diff_string1.length()==0 & diff_string2.length()==0){
                diff_string="";
            } else if (diff_string1.length()==0){
                diff_string = "-"+diff_string2;
            } else if (diff_string2.length()==0){
                diff_string = diff_string1;
            }else{
                diff_string = diff_string1+"-"+diff_string2;
            }
            
        }
        //中身が(...)*g(x)を含む
        else if(formula.find("("+blaket+")*")!=string::npos){
            size_t loc = formula.find("("+blaket+")*");
            string term1 = formula.replace(loc,blaket.length()+3,"");
            string term2 = "("+blaket+")";
            
            string diff_string1=partialDiff(term1,"x");
            if (diff_string1=="1") {
                diff_string1=term2;
            } else if (diff_string1!="") {
                diff_string1=term2+"*"+diff_string1;
            }
            string diff_string2=partialDiff(blaket,"x");
            index=0;
            do {
                index_bla=diff_string2.find("(",index);
                index_ket=diff_string2.find(")",index);
                index_plus=diff_string2.find("+",index);
                index_minus=diff_string2.find("-",index);
                
                if (blaketFloor==0) {
                    if (index_plus!=string::npos | index_minus!=string::npos) {
                        diff_string2="("+diff_string2+")";
                        break;
                    } else if(index_bla==string::npos) {
                        break;
                    }else{
                        blaketFloor++;
                        index = index_bla+1;
                        index_bla_sub = index_bla;
                    }
                } else {
                    if (index_bla < index_ket) {
                        blaketFloor++;
                        index = index_bla+1;
                    } else {
                        blaketFloor--;
                        index = index_ket+1;
                        index_ket_sub = index_ket;
                    }
                }
            } while (index<formula.length());
            if (diff_string2=="1") {
                diff_string2=term1;
            } else if (diff_string2!="") {
                diff_string2+="*"+term1;
            }
            
            if(diff_string1.length()==0 & diff_string2.length()==0){
                diff_string="";
            } else if (diff_string1.length()==0){
                diff_string = diff_string2;
            } else if (diff_string2.length()==0){
                diff_string = diff_string1;
            }else{
                diff_string = diff_string2+"+"+diff_string1;
            }
            
        }
        //中身が(...)/g(x)を含む
        else if(formula.find("("+blaket+")/")!=string::npos){
            size_t loc = formula.find("("+blaket+")/");
            string term1 = formula.replace(loc,blaket.length()+3,"");
            string term2 = "("+blaket+")";
            cout<<"term1: "<<term1<<endl;
            
            
            string diff_string1=partialDiff("1/"+term1,"x");
            cout<<diff_string1<<endl;
            if (diff_string1=="1") {
                diff_string1=term2;
            } else if (diff_string1!="") {
                diff_string1=term2+"*"+diff_string1;
            }
            string diff_string2=partialDiff(blaket,"x");
            index=0;
            do {
                index_bla=diff_string2.find("(",index);
                index_ket=diff_string2.find(")",index);
                index_plus=diff_string2.find("+",index);
                index_minus=diff_string2.find("-",index);
                
                if (blaketFloor==0) {
                    if (index_plus!=string::npos | index_minus!=string::npos) {
                        diff_string2="("+diff_string2+")";
                        break;
                    } else if(index_bla==string::npos) {
                        break;
                    }else{
                        blaketFloor++;
                        index = index_bla+1;
                        index_bla_sub = index_bla;
                    }
                } else {
                    if (index_bla < index_ket) {
                        blaketFloor++;
                        index = index_bla+1;
                    } else {
                        blaketFloor--;
                        index = index_ket+1;
                        index_ket_sub = index_ket;
                    }
                }
            } while (index<formula.length());
            if (diff_string2=="1") {
                diff_string2="1/"+term1;
            } else if (diff_string2!="") {
                diff_string2+="/"+term1;
            }
            
            if(diff_string1.length()==0 & diff_string2.length()==0){
                diff_string="";
            } else if (diff_string1.length()==0){
                diff_string = diff_string2;
            } else if (diff_string2.length()==0){
                diff_string = diff_string1;
            }else{
                diff_string = diff_string2+"+"+diff_string1;
            }
            
        }
        //中身がg(x)*xを含む
        else if(formula.find("*"+variable)!=string::npos){
            string replaceStr="*"+variable;
            size_t loc = formula.find(replaceStr);
            diff_string=formula.replace(loc,replaceStr.length(),"");
            
            string diff_string2=partialDiff(diff_string, variable);
            if (diff_string2!="") {
                diff_string+="+x*"+diff_string2;
            }
        }
        //中身がg(x)/xを含む
        else if(formula.find("/"+variable)!=string::npos){
            string replaceStr="/"+variable;
            size_t loc = formula.find(replaceStr);
            diff_string=formula.replace(loc,replaceStr.length(),"");
            
            string diff_string2=partialDiff(diff_string, variable);
            if (diff_string2!="") {
                diff_string=diff_string2+"/x-"+diff_string+"/x^2";
            } else{
                diff_string=diff_string+"/x^2";
            }
        }
        //中身がx*g(x)を含む
        else if(formula.find(variable+"*")!=string::npos){
            string replaceStr=variable+"*";
            size_t loc = formula.find(replaceStr);
            diff_string=formula.replace(loc,replaceStr.length(),"");
            
            string diff_string2=partialDiff(diff_string, variable);
            if (diff_string2!="") {
                diff_string+="+x*"+diff_string2;
            }
        }
        //中身がx/g(x)を含む
        else if(formula.find(variable+"/")!=string::npos){
            string replaceStr=variable+"/";
            size_t loc = formula.find(replaceStr);
            diff_string=formula.replace(loc,replaceStr.length(),"");
            
            string diff_string2=partialDiff("1/"+diff_string, variable);
            if (diff_string2!="") {
                diff_string="1/"+diff_string+"+x*"+diff_string2;
            } else {
                diff_string="1/"+diff_string;
            }
        }
        //中身がx
        else {
            diff_string="1";
        }
    }
    //中身に項が1つ以上, "+"による接続
    else if(index_plus<index_minus){
        string formula1,formula2;
        formula1=formula;
        formula2=formula;
        formula1.erase(index_plus,formula.length()-index_plus);
        formula2.erase(0,index_plus+1);
        //cout<<formula2;
        if (formula1.find("(")==0&formula1.find(")")==formula1.length()-1) {
            formula1.erase(0,1);
            formula1.erase(formula1.length()-1,1);
        }
        if (formula2.find("(")==0&formula2.find(")")==formula2.length()-1) {
            formula2.erase(0,1);
            formula2.erase(formula2.length()-1,1);
        }
        
        string diff_string1=partialDiff(formula1,variable);
        string diff_string2=partialDiff(formula2,variable);
        if(diff_string1.length()==0 & diff_string2.length()==0){
            diff_string="";
        } else if (diff_string1.length()==0){
            diff_string = diff_string2;
        } else if (diff_string2.length()==0){
            diff_string = diff_string1;
        }else{
            diff_string = diff_string1+"+"+diff_string2;
        }
        
    //中身に項が1つ以上, "-"による接続
    } else if(index_plus>index_minus){
        string formula1,formula2;
        formula1=formula;
        formula2=formula;
        formula1.erase(index_minus,formula.length()-index_minus);
        formula2.erase(0,index_minus+1);
        //cout<<formula2;
        if (formula1.find("(")==0&formula1.find(")")==formula1.length()-1) {
            formula1.erase(0,1);
            formula1.erase(formula1.length()-1,1);
        }
        if (formula2.find("(")==0&formula2.find(")")==formula2.length()-1) {
            formula2.erase(0,1);
            formula2.erase(formula2.length()-1,1);
        }
        
        string diff_string1=partialDiff(formula1,variable);
        string diff_string2=partialDiff(formula2,variable);
        if(diff_string1.length()==0 & diff_string2.length()==0){
            diff_string="";
        } else if (diff_string1.length()==0){
            diff_string = "-("+diff_string2+")";
        } else if (diff_string2.length()==0){
            diff_string = diff_string1;
        }else{
            diff_string = diff_string1+"-("+diff_string2+")";
        }
    }
    
    
    if(diff_string.find("ln(e)")!=string::npos) {
        diff_string.replace(diff_string.find("ln(e)"),4, "1");
    }
    if(diff_string.find("--")!=string::npos) {
        diff_string.replace(diff_string.find("--"),2, "+");
    }
    if(diff_string.find("+-")!=string::npos) {
        diff_string.replace(diff_string.find("+-"),2, "-");
    }
    if(diff_string.find("-+")!=string::npos) {
        diff_string.replace(diff_string.find("-+"),diff_string.find("-+")+2, "-");
    }
    if(diff_string.find("+")==0){
        diff_string.replace(0,1,"");
    }
    if(diff_string.find("*1")!=string::npos){
        diff_string.replace(diff_string.find("*1"),2,"");
    }
    if(diff_string.find("1*")!=string::npos){
        diff_string.replace(diff_string.find("1*"),2,"");
    }
    
    return diff_string;
}

int main(int argc, const char * argv[]) {
    
    string diff;
    diff= partialDiff("a*(b*x+c)^(a*x+1)*d", "x");
    if (diff.length()==0) {
        diff="0";
    }
    cout<<diff;
    
    
    return 0;
}
