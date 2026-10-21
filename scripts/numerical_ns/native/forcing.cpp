#include <SolverUtils/Forcing/Forcing.h>
#include <MultiRegions/ExpList.h>
#include <LocalRegions/TriExp.h>
#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <cstdlib>
#include "input_client.hpp"
namespace Nektar::SolverUtils {
class NSExactPublicForce : public Forcing {
    NSInputClient client_;
    int points_=0;
    std::vector<LocalRegions::TriExpSharedPtr> quadrature_;
public:
    NSExactPublicForce(const LibUtilities::SessionReaderSharedPtr& s,
                       const std::weak_ptr<EquationSystem>& e):Forcing(s,e){}
    static ForcingSharedPtr create(const LibUtilities::SessionReaderSharedPtr& s,
      const std::weak_ptr<EquationSystem>& e,
      const Array<OneD,MultiRegions::ExpListSharedPtr>& fields,
      const unsigned int& count,const TiXmlElement* node) {
        auto p=std::make_shared<NSExactPublicForce>(s,e);p->InitObject(fields,count,node);return p;
    }
    static std::string registration;
protected:
    void v_InitObject(const Array<OneD,MultiRegions::ExpListSharedPtr>& fields,
                      const unsigned int& count,const TiXmlElement*) override {
        if(count!=2) throw std::runtime_error("NS public force supports exactly two velocity fields");
        m_NumVariable=count;points_=fields[0]->GetTotPoints();
        std::vector<double> xy;
        const char* text=std::getenv("NS_REFERENCE_FORCE_QUADRATURE_FACTOR");
        const int factor=text?std::stoi(text):0;
        if(factor!=0 && (factor<2 || factor>4)) throw std::runtime_error("invalid force quadrature factor");
        if(factor) {
            // Integrate the public forcing from strictly interior Gauss points,
            // then L2-project onto the unchanged solver polynomial space. This
            // does not query a future solution or change any physical input.
            for(int e=0;e<fields[0]->GetExpSize();++e) {
                auto original=fields[0]->GetExp(e);
                auto geometry=std::dynamic_pointer_cast<SpatialDomains::Geometry2D>(original->GetGeom());
                if(!std::dynamic_pointer_cast<LocalRegions::TriExp>(original))
                    throw std::runtime_error("interior force integration requires triangle elements");
                int na=original->GetBasis(0)->GetNumModes(), nb=original->GetBasis(1)->GetNumModes();
                LibUtilities::BasisKey a(LibUtilities::eModified_A,na,
                    LibUtilities::PointsKey(factor*na,LibUtilities::eGaussGaussLegendre));
                LibUtilities::BasisKey b(LibUtilities::eModified_B,nb,
                    LibUtilities::PointsKey(factor*nb,LibUtilities::eGaussGaussLegendre));
                auto element=std::make_shared<LocalRegions::TriExp>(a,b,geometry);
                quadrature_.push_back(element);
                int n=element->GetTotPoints();Array<OneD,NekDouble> x(n),y(n),z(n);
                element->GetCoords(x,y,z);
                for(int i=0;i<n;++i){xy.push_back(x[i]);xy.push_back(y[i]);}
            }
        } else {
            Array<OneD,NekDouble> x(points_),y(points_),z(points_);
            fields[0]->GetCoords(x,y,z);xy.resize(2*points_);
            for(int i=0;i<points_;++i){xy[2*i]=x[i];xy[2*i+1]=y[i];}
        }
        client_.connect(xy);
        m_Forcing=Array<OneD,Array<OneD,NekDouble>>(2);
        for(int j=0;j<2;++j) m_Forcing[j]=Array<OneD,NekDouble>(points_,0.0);
    }
    void v_Apply(const Array<OneD,MultiRegions::ExpListSharedPtr>& fields,
                 const Array<OneD,Array<OneD,NekDouble>>&,
                 Array<OneD,Array<OneD,NekDouble>>& out,
                 const NekDouble& t) override {
        auto f=client_.query('F',t);
        if(quadrature_.empty()) {
            for(int j=0;j<2;++j) for(int i=0;i<points_;++i) m_Forcing[j][i]=f[2*i+j];
        } else {
            int source_offset=0;
            for(size_t e=0;e<quadrature_.size();++e) {
                auto quadrature=quadrature_[e], original=std::dynamic_pointer_cast<LocalRegions::TriExp>(fields[0]->GetExp(e));
                int n=quadrature->GetTotPoints(), offset=fields[0]->GetPhys_Offset(e);
                Array<OneD,NekDouble> values(n), coeffs(quadrature->GetNcoeffs()), result(original->GetTotPoints());
                for(int j=0;j<2;++j) {
                    for(int i=0;i<n;++i) values[i]=f[2*(source_offset+i)+j];
                    quadrature->FwdTrans(values,coeffs);original->BwdTrans(coeffs,result);
                    for(int i=0;i<original->GetTotPoints();++i) m_Forcing[j][offset+i]=result[i];
                }
                source_offset+=n;
            }
        }
        for(int j=0;j<2;++j) for(int i=0;i<points_;++i) out[j][i]+=m_Forcing[j][i];
    }
};
std::string NSExactPublicForce::registration=GetForcingFactory().RegisterCreatorFunction(
    "NSExactPublicInput",NSExactPublicForce::create,"Controller public u0/f bridge, no future solution access");
}
