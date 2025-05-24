const models = require('../model/producto.Model');

const  model= new models();

class ProductoService { 
  
    constructor() {}

    async find(idproducto) {
     //console.log('ingereso');
      const res = await model.findOne(idproducto);
      //console.log( res );
      return res;
    }

    async findDescripcion(descripcion) {
      //console.log('ingereso');
       const res = await model.findDescripcion(descripcion);
      // console.log( res );
       return res;
     }

     async findUbicacion(ubicacion) {
       const res = await model.findUbicacion(ubicacion);
       return res;
     }

    async findOne(id) {
      const res = await models.findByPk(id);
      return res;
    }

    async create(data) {
      const res = await models.create(data);
      return res;
    }

    async update(id, data) {
      const model = await this.findOne(id);
      const res = await model.update(data);
      return res;
    }

    async delete(id) {
      const model = await this.findOne(id);
      await model.destroy();
      return { deleted: true };
    }
  
  }
  
  module.exports = ProductoService;