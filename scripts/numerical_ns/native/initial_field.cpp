#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/FieldIO.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraphIO.h>
#include "input_client.hpp"
#include <iostream>
int main(int argc,char** argv) {
    using namespace Nektar;
    try {
        auto session=LibUtilities::SessionReader::CreateInstance(argc,argv);
        auto graph=SpatialDomains::MeshGraphIO::Read(session);
        auto field=MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(session,graph,"u");
        int n=field->GetTotPoints();
        Array<OneD,NekDouble> x(n),y(n),z(n);field->GetCoords(x,y,z);
        std::vector<double> xy(2*n);for(int i=0;i<n;++i){xy[2*i]=x[i];xy[2*i+1]=y[i];}
        NSInputClient client;client.connect(xy);
        double t0=0;session->LoadParameter("Time",t0,0.0);auto uv=client.query('I',t0);
        auto definitions=field->GetFieldDefinitions();
        std::vector<std::vector<NekDouble>> data(definitions.size());
        for(auto& def:definitions) def->m_fields={"u","v","p"};
        for(int component=0;component<3;++component) {
            Array<OneD,NekDouble> values(n,0.0),coeff(field->GetNcoeffs(),0.0);
            if(component<2)for(int i=0;i<n;++i)values[i]=uv[2*i+component];
            field->FwdTrans(values,coeff);
            for(size_t j=0;j<definitions.size();++j)field->AppendFieldData(definitions[j],data[j],coeff);
        }
        LibUtilities::Write("initial.fld",definitions,data);
        std::cout<<"Public initial velocity projected at "<<n<<" quadrature points\n";
        session->Finalise();return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
