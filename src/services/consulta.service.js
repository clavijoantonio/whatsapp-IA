
const apiService = require('./producto.service.js')
const service = new apiService();

class ConsultaService { 
  
    constructor() {}

    async findid(idproducto) {
     //console.log('ingereso');
      const res = await service.find(idproducto);
      //const dato=JSON.stringify(res)
     // console.log( res );
      return res;
    }

      async consultaDescripcion (mensajeRecibido)  {
        try {
          console.log(mensajeRecibido);
          console.log('ingreso a la consulta');
          const data = await service.findDescripcion(mensajeRecibido);
         // const dato=JSON.stringify(data)
          //console.log(dato)
        return data
        } catch (error) {
          console.error('Error en init:', error);
        }
  
      };

      async consultaUbicacion (mensajeRecibido)  {
        try {
          console.log(mensajeRecibido);
          const data = await service.findUbicacion(mensajeRecibido);
        // const dato=JSON.stringify(data)
          console.log(data)
        return data
        } catch (error) {
          console.error('Error en init:', error);
        }
  
      };
       
       
}
module.exports = ConsultaService;