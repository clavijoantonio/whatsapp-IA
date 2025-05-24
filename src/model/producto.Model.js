
const connection = require('../config/config');

class Producto  {
    tableName = 'producto';

      async find (params={}) {
        let sql = `SELECT * FROM ${this.tableName}`;
        
        const res= connection.query(sql)
        console.log(res);
        return await res;

    }
    
    async findDescripcion (descripcion) {
        
       /* console.log(descripcion);
        let sql = `SELECT * FROM ${this.tableName} where descripcion like ${descripcion}`;
        console.log(sql);
        const res= connection.query(sql)*/
        // Consulta segura usando placeholders (?)
        descripcion=descripcion.trim();

    const sql = `SELECT * FROM ?? WHERE descripcion LIKE ? `;
    const parametros = [this.tableName, `%${descripcion}%`];

    // Ejecutar la consulta
    const res = await connection.query(sql, parametros);

       console.log(res);
        return await res;
    }

    async findUbicacion (ubicacion) {
        
         ubicacion=ubicacion.trim();
 
     const sql = `SELECT idproducto,tipoInmueble FROM ?? WHERE ubicacion LIKE ? `;
     const parametros = [this.tableName, `%${ubicacion}%`];
 
     // Ejecutar la consulta
     const res = await connection.query(sql, parametros);
 
        console.log(res);
         return await res;
     }

      async findOne (idproducto)  {
       // idproducto=idproducto;
       console.log(idproducto);
        const sql = `SELECT * FROM ?? WHERE idproducto LIKE ? `;
        const parametros = [this.tableName, idproducto];
        // Ejecutar la consulta
        const res = await connection.query(sql, parametros);
           //console.log(res);
            return await res;
    }

    create = async ({ username, password, first_name, last_name, email, role = Role.SuperUser, age = 0 }) => {
        const sql = `INSERT INTO ${this.tableName}
        (username, password, first_name, last_name, email, role, age) VALUES (?,?,?,?,?,?,?)`;

        const result = await query(sql, [username, password, first_name, last_name, email, role, age]);
        const affectedRows = result ? result.affectedRows : 0;

        return affectedRows;
    }

    update = async (params, id) => {
        const { columnSet, values } = multipleColumnSet(params)

        const sql = `UPDATE user SET ${columnSet} WHERE id = ?`;

        const result = await query(sql, [...values, id]);

        return result;
    }

    delete = async (id) => {
        const sql = `DELETE FROM ${this.tableName}
        WHERE id = ?`;
        const result = await query(sql, [id]);
        const affectedRows = result ? result.affectedRows : 0;

        return affectedRows;
    }
}
module.exports = Producto;