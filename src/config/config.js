/*require('dotenv').config();

const dotenv = require("dotenv");
dotenv.config();

const mysql = require('mysql');
let connection;

try {
    connection = mysql.createConnection({
        host: process.env.DBHOST,
        user: process.env.DBUSER,
        password: process.env.DBPASS,
        database: process.env.DBNAME
    }); 
    
} catch (error) {
    console.log("Error al conectar con la base de datos");
}

module.exports = {connection};*/
require('dotenv').config();

const mysql = require('mysql');
const util = require('util'); // Utilizar util.promisify para convertir funciones de callback a Promises

let connection;

try {
    connection = mysql.createConnection({
        host: process.env.DB_HOST,
        user: process.env.DB_USER,
        password: process.env.DB_PASSWORD,
        database: process.env.DB_NAME
    }); 
    connection.connect((err) => {
        if (err) {
            console.error('Error connecting to the database:', err.stack);
            return;
        }
        console.log('Connected to the database as id', connection.threadId);
    });
} catch (error) {
    console.error("Error al conectar con la base de datos:", error);
}

const query = util.promisify(connection.query).bind(connection);
/*(async () => {
    try {
        const rows = await query('SELECT COUNT (*)  FROM producto');
        console.log('The solution is:',rows); // Debería imprimir "The solution is: 2"
    } catch (error) {
        console.error('Error executing test query:', error);
    }
})();*/

module.exports = {query};