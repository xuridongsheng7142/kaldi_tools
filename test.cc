#include <map>
#include <iostream>
#include <string>
using namespace std;
int main(){
map<string, string> namemap;
namemap["岳不群"]="华山派掌门人，人称君子剑";
namemap["张三丰"]="武当掌门人，太极拳创始人";
namemap["东方不败"]="第一高手，葵花宝典";
if(namemap.find("岳不群") != namemap.end()){
    cout << "haha" << endl;
}
}
