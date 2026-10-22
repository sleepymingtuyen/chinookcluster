class InterestDAO:
    """Data Access Layer: gom các câu SQL lấy dữ liệu 'sự quan tâm' của khách hàng."""

    @staticmethod
    def get_purchases(conn):
        """Mỗi dòng = một dòng hoá đơn (khách nào mua track nào, thuộc album/nghệ sĩ/thể loại nào)."""
        sql = """
        SELECT i.CustomerId,
               il.TrackId,
               il.Quantity,
               il.UnitPrice,
               il.Quantity * il.UnitPrice AS LineTotal,
               t.Name        AS TrackName,
               t.AlbumId,
               al.Title      AS AlbumTitle,
               al.ArtistId,
               ar.Name       AS ArtistName,
               t.GenreId,
               g.Name        AS GenreName
        FROM InvoiceLine il
        JOIN Invoice i      ON i.InvoiceId = il.InvoiceId
        JOIN Track t        ON t.TrackId   = il.TrackId
        LEFT JOIN Album al  ON al.AlbumId  = t.AlbumId
        LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
        LEFT JOIN Genre g   ON g.GenreId   = t.GenreId
        """
        return conn.queryDataset(sql)

    @staticmethod
    def get_customers(conn):
        sql = """
        SELECT CustomerId, FirstName || ' ' || LastName AS FullName, Country
        FROM Customer
        ORDER BY CustomerId
        """
        return conn.queryDataset(sql)

    @staticmethod
    def get_album_genre(conn):
        """Thể loại chủ đạo của từng album (thể loại có nhiều track nhất trong album).
        Trả về Series: index = AlbumId, value = tên thể loại."""
        sql = """
        SELECT t.AlbumId, g.Name AS GenreName, COUNT(*) AS n
        FROM Track t
        JOIN Genre g ON g.GenreId = t.GenreId
        WHERE t.AlbumId IS NOT NULL
        GROUP BY t.AlbumId, g.Name
        """
        df = conn.queryDataset(sql)
        df = (df.sort_values(["AlbumId", "n", "GenreName"], ascending=[True, False, True])
                .drop_duplicates("AlbumId"))
        return df.set_index("AlbumId")["GenreName"]
