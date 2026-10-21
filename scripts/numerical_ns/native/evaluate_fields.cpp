// Read precomputed Nektar coefficient snapshots. This binary has no Oracle.
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/FieldIO.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <iostream>
struct Request {int id; double x,y;};
int main(int argc,char** argv) {
 using namespace Nektar;
 try {
  auto session=LibUtilities::SessionReader::CreateInstance(argc,argv);
  auto graph=SpatialDomains::MeshGraphIO::Read(session);
  auto field=MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(session,graph,"u");
  std::ifstream in("evaluation_requests.txt");
  std::map<std::string,std::vector<Request>> groups;
  int id;std::string name;double x,y;
  while(in>>id>>name>>x>>y) groups[name].push_back({id,x,y});
  if(groups.empty())throw std::runtime_error("no post-solve queries");
  std::ofstream out("evaluation_values.txt");out<<std::setprecision(17);
  for(auto &group:groups) {
   std::vector<LibUtilities::FieldDefinitionsSharedPtr> defs;
   std::vector<std::vector<NekDouble>> data;
   LibUtilities::FieldMetaDataMap metadata;
   LibUtilities::Import(group.first,defs,data,metadata);
   std::vector<std::vector<double>> values(2,std::vector<double>(group.second.size()));
   for(int c=0;c<2;++c) {
    Array<OneD,NekDouble> coeff(field->GetNcoeffs(),0.0),phys(field->GetTotPoints(),0.0);
    std::string component=c==0?"u":"v";
    for(size_t k=0;k<defs.size();++k)field->ExtractDataToCoeffs(defs[k],data[k],component,coeff);
    field->BwdTrans(coeff,phys);
    for(size_t k=0;k<group.second.size();++k) {
     auto r=group.second[k];Array<OneD,NekDouble> coord(2,0.0);coord[0]=r.x;coord[1]=r.y;
     Array<OneD,NekDouble> local(2,0.0);
     int element=field->GetExpIndex(coord,local,1e-10);
     if(element<0)throw std::runtime_error("point outside spectral mesh");
     Array<OneD,const NekDouble> element_phys=phys+field->GetPhys_Offset(element);
     values[c][k]=field->GetExp(element)->StdPhysEvaluate(local,element_phys);
    }
   }
   for(size_t k=0;k<group.second.size();++k)
    out<<group.second[k].id<<" "<<values[0][k]<<" "<<values[1][k]<<"\n";
  }
  session->Finalise();return 0;
 }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
